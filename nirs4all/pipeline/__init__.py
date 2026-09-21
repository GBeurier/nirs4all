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

# Keep package initialization free of child imports: API and storage clients
# may import separate pipeline modules concurrently during application startup.
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from .engine import DualRunMismatchError, DualRunUnsupported
    from .explainer import Explainer
    from .keyword_registry import (
        KEYWORD_REGISTRY_SCHEMA_ID,
        KEYWORD_REGISTRY_SCHEMA_VERSION,
        KEYWORD_REGISTRY_VERSION,
        get_keyword_registry,
        get_keyword_registry_schema,
        keyword_registry_json,
        keyword_registry_schema_json,
    )
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

_LAZY_EXPORTS = {
    'BundleFormat': ('.bundle', 'BundleFormat'),
    'BundleGenerator': ('.bundle', 'BundleGenerator'),
    'BundleLoader': ('.bundle', 'BundleLoader'),
    'BundleMetadata': ('.bundle', 'BundleMetadata'),
    'PipelineConfigs': ('.config', 'PipelineConfigs'),
    'ArtifactProvider': ('.config.context', 'ArtifactProvider'),
    'LoaderArtifactProvider': ('.config.context', 'LoaderArtifactProvider'),
    'MapArtifactProvider': ('.config.context', 'MapArtifactProvider'),
    'DualRunMismatchError': ('.engine', 'DualRunMismatchError'),
    'DualRunUnsupported': ('.engine', 'DualRunUnsupported'),
    'Explainer': ('.explainer', 'Explainer'),
    'KEYWORD_REGISTRY_SCHEMA_ID': ('.keyword_registry', 'KEYWORD_REGISTRY_SCHEMA_ID'),
    'KEYWORD_REGISTRY_SCHEMA_VERSION': ('.keyword_registry', 'KEYWORD_REGISTRY_SCHEMA_VERSION'),
    'KEYWORD_REGISTRY_VERSION': ('.keyword_registry', 'KEYWORD_REGISTRY_VERSION'),
    'get_keyword_registry': ('.keyword_registry', 'get_keyword_registry'),
    'get_keyword_registry_schema': ('.keyword_registry', 'get_keyword_registry_schema'),
    'keyword_registry_json': ('.keyword_registry', 'keyword_registry_json'),
    'keyword_registry_schema_json': ('.keyword_registry', 'keyword_registry_schema_json'),
    'MinimalArtifactProvider': ('.minimal_predictor', 'MinimalArtifactProvider'),
    'MinimalPredictor': ('.minimal_predictor', 'MinimalPredictor'),
    'Predictor': ('.predictor', 'Predictor'),
    'FoldStrategy': ('.resolver', 'FoldStrategy'),
    'PredictionResolver': ('.resolver', 'PredictionResolver'),
    'ResolvedPrediction': ('.resolver', 'ResolvedPrediction'),
    'SourceType': ('.resolver', 'SourceType'),
    'ExtractedPipeline': ('.retrainer', 'ExtractedPipeline'),
    'RetrainArtifactProvider': ('.retrainer', 'RetrainArtifactProvider'),
    'RetrainConfig': ('.retrainer', 'RetrainConfig'),
    'Retrainer': ('.retrainer', 'Retrainer'),
    'RetrainMode': ('.retrainer', 'RetrainMode'),
    'StepMode': ('.retrainer', 'StepMode'),
    'DatasetInfo': ('.run', 'DatasetInfo'),
    'Run': ('.run', 'Run'),
    'RunConfig': ('.run', 'RunConfig'),
    'RunStatus': ('.run', 'RunStatus'),
    'RunSummary': ('.run', 'RunSummary'),
    'TemplateInfo': ('.run', 'TemplateInfo'),
    'generate_run_id': ('.run', 'generate_run_id'),
    'get_metric_info': ('.run', 'get_metric_info'),
    'is_better_score': ('.run', 'is_better_score'),
    'PipelineRunner': ('.runner', 'PipelineRunner'),
    'PipelineLibrary': ('.storage.library', 'PipelineLibrary'),
    'ExecutionStep': ('.trace', 'ExecutionStep'),
    'ExecutionTrace': ('.trace', 'ExecutionTrace'),
    'MinimalPipeline': ('.trace', 'MinimalPipeline'),
    'MinimalPipelineStep': ('.trace', 'MinimalPipelineStep'),
    'StepArtifacts': ('.trace', 'StepArtifacts'),
    'TraceBasedExtractor': ('.trace', 'TraceBasedExtractor'),
    'TraceRecorder': ('.trace', 'TraceRecorder'),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "PipelineConfigs",
    "PipelineRunner",
    "DualRunUnsupported",
    "DualRunMismatchError",
    "Predictor",
    "Explainer",
    "KEYWORD_REGISTRY_SCHEMA_ID",
    "KEYWORD_REGISTRY_SCHEMA_VERSION",
    "KEYWORD_REGISTRY_VERSION",
    "get_keyword_registry",
    "get_keyword_registry_schema",
    "keyword_registry_json",
    "keyword_registry_schema_json",
    "PredictionResolver",
    "ResolvedPrediction",
    "SourceType",
    "FoldStrategy",
    "ArtifactProvider",
    "MapArtifactProvider",
    "LoaderArtifactProvider",
    "PipelineLibrary",
    # Phase 5: Minimal Pipeline Execution
    "TraceBasedExtractor",
    "MinimalPipeline",
    "MinimalPipelineStep",
    "MinimalPredictor",
    "MinimalArtifactProvider",
    "ExecutionTrace",
    "ExecutionStep",
    "StepArtifacts",
    "TraceRecorder",
    # Phase 6: Bundle Export
    "BundleGenerator",
    "BundleLoader",
    "BundleFormat",
    "BundleMetadata",
    # Phase 7: Retrain & Transfer
    "Retrainer",
    "RetrainMode",
    "StepMode",
    "RetrainConfig",
    "ExtractedPipeline",
    "RetrainArtifactProvider",
    # Run entities
    "Run",
    "RunStatus",
    "RunConfig",
    "RunSummary",
    "TemplateInfo",
    "DatasetInfo",
    "generate_run_id",
    "get_metric_info",
    "is_better_score",
]
