"""
Base parser interface for dataset configuration.

This module defines the abstract base class for configuration parsers.
All parsers should inherit from BaseParser and implement the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ParserResult:
    """Result of parsing a configuration.

    Attributes:
        success: Whether parsing was successful.
        config: The parsed configuration dictionary.
        dataset_name: The extracted or inferred dataset name.
        errors: List of error messages if parsing failed.
        warnings: List of warning messages (non-fatal issues).
        source_type: Type of source that was parsed ('dict', 'file', 'folder', 'array').
    """

    success: bool
    config: Optional[Dict[str, Any]] = None
    dataset_name: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_type: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return f"ParserResult(success=True, name='{self.dataset_name}')"
        return f"ParserResult(success=False, errors={self.errors})"


class BaseParser(ABC):
    """Abstract base class for configuration parsers.

    Subclasses must implement:
    - can_parse(): Check if this parser can handle the input
    - parse(): Parse the input and return a ParserResult
    """

    @abstractmethod
    def can_parse(self, input_data: Any) -> bool:
        """Check if this parser can handle the given input.

        Args:
            input_data: The input to check.

        Returns:
            True if this parser can handle the input, False otherwise.
        """
        pass

    @abstractmethod
    def parse(self, input_data: Any) -> ParserResult:
        """Parse the input and return a configuration.

        Args:
            input_data: The input to parse.

        Returns:
            ParserResult with parsed configuration or errors.
        """
        pass

    def _extract_name_from_path(self, path: Union[str, Path]) -> str:
        """Extract a dataset name from a file or folder path.

        Args:
            path: Path to extract name from.

        Returns:
            Cleaned dataset name.
        """
        path_obj = Path(path)

        # For files, use stem (filename without extension)
        if path_obj.is_file() or path_obj.suffix:
            return self._clean_name(path_obj.stem)

        # For folders, use the folder name
        return self._clean_name(path_obj.name)

    def _clean_name(self, name: str) -> str:
        """Clean a name to be a valid dataset identifier.

        Args:
            name: Raw name to clean.

        Returns:
            Cleaned name with only alphanumeric and underscore characters.
        """
        # Replace non-alphanumeric with underscore
        cleaned = ''.join(c if c.isalnum() else '_' for c in name)
        # Remove consecutive underscores
        while '__' in cleaned:
            cleaned = cleaned.replace('__', '_')
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Ensure lowercase
        cleaned = cleaned.lower()
        # Ensure non-empty
        return cleaned or "unnamed_dataset"
