"""
Sample filtering operators for nirs4all.

This module provides operators for filtering (excluding) samples from training datasets.
Filters are non-destructive - they mark samples as excluded in the indexer rather than
removing data.

Classes:
    SampleFilter: Base class for all sample filtering operators
    CompositeFilter: Combine multiple filters with AND/OR logic
    YOutlierFilter: Filter samples with outlier target values (IQR, zscore, percentile, MAD)
    XOutlierFilter: Filter samples with outlier spectral features (Mahalanobis, PCA, LOF, etc.)
    SpectralQualityFilter: Filter samples with poor spectral quality (NaN, zeros, variance)
    HighLeverageFilter: Filter high-leverage samples that may unduly influence models
    MetadataFilter: Filter samples based on metadata column values
    FilteringReport: Comprehensive report of sample filtering operations
    FilteringReportGenerator: Generator for creating filtering reports

Example:
    >>> from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter
    >>>
    >>> # In a pipeline
    >>> pipeline = [
    ...     {
    ...         "sample_filter": {
    ...             "filters": [
    ...                 YOutlierFilter(method="iqr", threshold=1.5),
    ...                 XOutlierFilter(method="mahalanobis", threshold=3.0),
    ...             ],
    ...             "report": True,
    ...         }
    ...     },
    ...     "snv",
    ...     "model:PLSRegression",
    ... ]
"""

from .base import SampleFilter, CompositeFilter
from .y_outlier import YOutlierFilter
from .x_outlier import XOutlierFilter
from .spectral_quality import SpectralQualityFilter
from .high_leverage import HighLeverageFilter
from .metadata import MetadataFilter
from .report import FilteringReport, FilteringReportGenerator, FilterResult

__all__ = [
    "SampleFilter",
    "CompositeFilter",
    "YOutlierFilter",
    "XOutlierFilter",
    "SpectralQualityFilter",
    "HighLeverageFilter",
    "MetadataFilter",
    "FilteringReport",
    "FilteringReportGenerator",
    "FilterResult",
]
