"""
Parsers module for dataset configuration.

This module provides parsers for converting various input formats
to the normalized DatasetConfigSchema.

Parsers:
- LegacyParser: Handles the current train_x/test_x format
- FilesParser: Handles the new files syntax
- SourcesParser: Handles multi-source datasets (Phase 6)
- VariationsParser: Handles feature variations / preprocessed data (Phase 7)
- FolderParser: Handles folder auto-scanning

The ConfigNormalizer combines all parsers and produces a canonical representation.
"""

from .base import BaseParser, ParserResult
from .legacy_parser import LegacyParser
from .files_parser import FilesParser, SourcesParser, VariationsParser
from .folder_parser import FolderParser
from .normalizer import ConfigNormalizer, normalize_config

__all__ = [
    "BaseParser",
    "ParserResult",
    "LegacyParser",
    "FilesParser",
    "SourcesParser",
    "VariationsParser",
    "FolderParser",
    "ConfigNormalizer",
    "normalize_config",
]
