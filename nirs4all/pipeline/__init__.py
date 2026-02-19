"""
Pipeline module for nirs4all package.

This module contains pipeline classes for processing workflows.

Phase 5 Additions:
    - TraceBasedExtractor: Extracts minimal pipeline from execution trace
    - MinimalPipeline: Minimal pipeline ready for prediction replay
    - MinimalPipelineStep: A single step in the minimal pipeline
    - MinimalPredictor: Executes minimal pipeline for efficient prediction
    - MinimalArtifactProvider: Provides artifacts from MinimalPipeline

Phase 6 Additions:
    - BundleGenerator: Creates standalone prediction bundles (.n4a, .n4a.py)
    - BundleLoader: Loads and predicts from exported bundles
    - BundleFormat: Enumeration of supported bundle formats
    - BundleMetadata: Bundle metadata structure

Phase 7 Additions:
    - Retrainer: Handles retraining with full/transfer/finetune modes
    - RetrainMode: Enumeration of retrain modes
    - StepMode: Per-step mode override for fine-grained control
    - ExtractedPipeline: Extracted pipeline for inspection/modification
    - RetrainArtifactProvider: Artifact provider respecting retrain modes
"""

from .bundle import (
    BundleFormat,
    BundleGenerator,
    BundleLoader,
    BundleMetadata,
)
from .config import PipelineConfigs
from .config.context import (
    ArtifactProvider,
    LoaderArtifactProvider,
    MapArtifactProvider,
)
from .explainer import Explainer
from .minimal_predictor import MinimalArtifactProvider, MinimalPredictor
from .predictor import Predictor
from .resolver import FoldStrategy, PredictionResolver, ResolvedPrediction, SourceType
from .retrainer import (
    ExtractedPipeline,
    RetrainArtifactProvider,
    RetrainConfig,
    Retrainer,
    RetrainMode,
    StepMode,
)
from .run import (
    DatasetInfo,
    Run,
    RunConfig,
    RunStatus,
    RunSummary,
    TemplateInfo,
    generate_run_id,
    get_metric_info,
    is_better_score,
)
from .runner import PipelineRunner
from .storage.library import PipelineLibrary
from .trace import (
    ExecutionStep,
    ExecutionTrace,
    MinimalPipeline,
    MinimalPipelineStep,
    StepArtifacts,
    TraceBasedExtractor,
    TraceRecorder,
)

__all__ = [
    'PipelineConfigs',
    'PipelineRunner',
    'Predictor',
    'Explainer',
    'PredictionResolver',
    'ResolvedPrediction',
    'SourceType',
    'FoldStrategy',
    'ArtifactProvider',
    'MapArtifactProvider',
    'LoaderArtifactProvider',
    'PipelineLibrary',
    # Phase 5: Minimal Pipeline Execution
    'TraceBasedExtractor',
    'MinimalPipeline',
    'MinimalPipelineStep',
    'MinimalPredictor',
    'MinimalArtifactProvider',
    'ExecutionTrace',
    'ExecutionStep',
    'StepArtifacts',
    'TraceRecorder',
    # Phase 6: Bundle Export
    'BundleGenerator',
    'BundleLoader',
    'BundleFormat',
    'BundleMetadata',
    # Phase 7: Retrain & Transfer
    'Retrainer',
    'RetrainMode',
    'StepMode',
    'RetrainConfig',
    'ExtractedPipeline',
    'RetrainArtifactProvider',
    # Run entities
    'Run',
    'RunStatus',
    'RunConfig',
    'RunSummary',
    'TemplateInfo',
    'DatasetInfo',
    'generate_run_id',
    'get_metric_info',
    'is_better_score',
]
