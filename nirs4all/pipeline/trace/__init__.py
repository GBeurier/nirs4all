"""
Execution Trace module for nirs4all pipeline (V3).

This module provides data structures and utilities for recording the exact
execution path through a pipeline, enabling deterministic prediction replay.

V3 improvements:
- OperatorChain tracking for complete execution path
- Branch and source indexes for artifact lookup
- Chain-based artifact identification
- Proper recording of branch substeps

Key Components:
    - ExecutionTrace: Complete trace of a pipeline execution path
    - ExecutionStep: Record of a single step's execution with chain tracking
    - StepArtifacts: Artifacts produced by a single step with V3 indexes
    - TraceRecorder: Records traces with chain/branch stacks
    - TraceBasedExtractor: Extracts minimal pipeline from trace
    - MinimalPipeline: Minimal pipeline ready for prediction replay

Design Principles:
    1. Controller-Agnostic: Works with any controller type
    2. Deterministic: Same chain -> same artifacts
    3. Complete: Full execution path tracking
    4. Composable: Same infrastructure for predict, retrain, transfer, export

Usage:
    >>> from nirs4all.pipeline.trace import TraceRecorder, ExecutionTrace
    >>>
    >>> # During training with V3 chain tracking
    >>> recorder = TraceRecorder(pipeline_uid="0001_pls_abc123")
    >>> recorder.start_step(step_index=1, operator_type="transform", operator_class="SNV")
    >>> chain = recorder.build_chain_for_artifact(1, "SNV")
    >>> recorder.record_artifact("0001$abc123:all", chain_path=chain.to_path())
    >>> recorder.end_step()
    >>> trace = recorder.finalize()
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
