"""
Bundle module for nirs4all pipeline exports.

This module provides functionality for exporting trained pipelines as standalone
prediction bundles that can be used for deployment, sharing, or archival.

Supported Formats:
    - .n4a: Full bundle (ZIP archive with artifacts and metadata)
    - .n4a.py: Portable Python script with embedded artifacts (base64)

Key Components:
    - BundleGenerator: Creates prediction bundles from trained pipelines
    - BundleLoader: Loads bundles for prediction or inspection
    - BundleMetadata: Bundle metadata and manifest structure
    - PortablePredictor: Standalone predictor from .n4a.py bundle

Usage (Export):
    >>> from nirs4all.pipeline import PipelineRunner
    >>>
    >>> runner = PipelineRunner()
    >>> predictions, _ = runner.run(pipeline, dataset)
    >>> best_pred = predictions.top(n=1)[0]
    >>>
    >>> # Export to .n4a bundle
    >>> runner.export(best_pred, "exports/wheat_model.n4a")
    >>>
    >>> # Export to portable Python script
    >>> runner.export(best_pred, "exports/wheat_model.n4a.py", format='n4a.py')

Usage (Predict from bundle):
    >>> from nirs4all.pipeline import PipelineRunner
    >>>
    >>> runner = PipelineRunner()
    >>> y_pred, preds = runner.predict("exports/wheat_model.n4a", X_new)
    >>>
    >>> # Or using standalone script (no nirs4all dependency)
    >>> # python wheat_model.n4a.py input.csv output.csv

Design Principles:
    1. Self-Contained: Bundle includes all artifacts and metadata
    2. Portable: Minimal dependencies for prediction
    3. Versioned: Bundle format is versioned for compatibility
    4. Deterministic: Same bundle -> same predictions
"""

from nirs4all.pipeline.bundle.generator import (
    BundleGenerator,
    BundleFormat,
)
from nirs4all.pipeline.bundle.loader import (
    BundleLoader,
    BundleMetadata,
)

__all__ = [
    "BundleGenerator",
    "BundleFormat",
    "BundleLoader",
    "BundleMetadata",
]
