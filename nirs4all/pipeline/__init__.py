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

from .config import PipelineConfigs
from .runner import PipelineRunner
from .predictor import Predictor
from .explainer import Explainer
from .storage.io_writer import PipelineWriter
from .storage.io_exporter import WorkspaceExporter
from .storage.io_resolver import TargetResolver
from .storage.library import PipelineLibrary
from .resolver import PredictionResolver, ResolvedPrediction, SourceType, FoldStrategy
from .config.context import (
    ArtifactProvider,
    MapArtifactProvider,
    LoaderArtifactProvider,
)
from .trace import (
    TraceBasedExtractor,
    MinimalPipeline,
    MinimalPipelineStep,
    ExecutionTrace,
    ExecutionStep,
    StepArtifacts,
    TraceRecorder,
)
from .minimal_predictor import MinimalPredictor, MinimalArtifactProvider
from .bundle import (
    BundleGenerator,
    BundleLoader,
    BundleFormat,
    BundleMetadata,
)
from .retrainer import (
    Retrainer,
    RetrainMode,
    StepMode,
    RetrainConfig,
    ExtractedPipeline,
    RetrainArtifactProvider,
)

__all__ = [
    'PipelineConfigs',
    'PipelineRunner',
    'Predictor',
    'Explainer',
    'PipelineWriter',
    'WorkspaceExporter',
    'TargetResolver',
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
]
