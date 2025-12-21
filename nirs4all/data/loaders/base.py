"""
Base file loader interface and registry.

This module defines the abstract FileLoader base class and LoaderRegistry
for a pluggable file loading system. It supports multiple file formats
with automatic format detection and configurable loading parameters.

Phase 2 Implementation - Dataset Configuration Roadmap
"""

import gzip
import io
import tarfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import pandas as pd


class LoaderError(Exception):
    """Base exception for loader errors."""
    pass


class FormatNotSupportedError(LoaderError):
    """Raised when a file format is not supported."""
    pass


class FileLoadError(LoaderError):
    """Raised when a file cannot be loaded."""
    pass


class LoaderResult:
    """Result container for file loading operations.

    Attributes:
        data: The loaded data as a pandas DataFrame.
        report: Dictionary containing loading metadata and diagnostics.
        na_mask: Boolean Series indicating rows with NA values.
        headers: List of column headers.
        header_unit: The unit type for headers (e.g., 'cm-1', 'nm').
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        report: Optional[Dict[str, Any]] = None,
        na_mask: Optional[pd.Series] = None,
        headers: Optional[List[str]] = None,
        header_unit: str = "cm-1",
    ):
        self.data = data
        self.report = report or {}
        self.na_mask = na_mask
        self.headers = headers or []
        self.header_unit = header_unit

    @property
    def success(self) -> bool:
        """Check if loading was successful."""
        return self.data is not None and self.report.get("error") is None

    @property
    def error(self) -> Optional[str]:
        """Get error message if loading failed."""
        return self.report.get("error")


class FileLoader(ABC):
    """Abstract base class for file loaders.

    All file format loaders should inherit from this class and implement
    the required methods for loading and format detection.

    Class Attributes:
        supported_extensions: Tuple of file extensions this loader handles.
        name: Human-readable name for the loader.
        priority: Loading priority (lower = higher priority) when multiple
            loaders match. Default: 50.

    Example:
        >>> class CSVLoader(FileLoader):
        ...     supported_extensions = (".csv",)
        ...     name = "CSV Loader"
        ...
        ...     @classmethod
        ...     def supports(cls, path: Path) -> bool:
        ...         return path.suffix.lower() in cls.supported_extensions
        ...
        ...     def load(self, path: Path, **params) -> LoaderResult:
        ...         # Load CSV file
        ...         pass
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = ()
    name: ClassVar[str] = "Base Loader"
    priority: ClassVar[int] = 50

    @classmethod
    @abstractmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader can handle the given file.

        Args:
            path: Path to the file to check.

        Returns:
            True if this loader can handle the file, False otherwise.
        """
        pass

    @abstractmethod
    def load(
        self,
        path: Path,
        **params: Any,
    ) -> LoaderResult:
        """Load data from a file.

        Args:
            path: Path to the file to load.
            **params: Loader-specific parameters.

        Returns:
            LoaderResult containing the loaded data and metadata.

        Raises:
            FileLoadError: If the file cannot be loaded.
        """
        pass

    @classmethod
    def detect_format(cls, path: Path) -> Optional[str]:
        """Detect the file format from the path.

        Args:
            path: Path to analyze.

        Returns:
            Format name if detected, None otherwise.
        """
        suffix = path.suffix.lower()
        if suffix in cls.supported_extensions:
            return cls.name
        return None

    @classmethod
    def get_base_path(cls, path: Path) -> Path:
        """Get the base path without compression extensions.

        For example, 'data.csv.gz' -> 'data.csv'

        Args:
            path: Path to process.

        Returns:
            Path without compression extension(s).
        """
        compression_suffixes = {".gz", ".gzip", ".zip", ".bz2", ".xz"}
        result = path
        while result.suffix.lower() in compression_suffixes:
            result = result.with_suffix("")
        return result


class ArchiveHandler:
    """Utility class for handling compressed files and archives.

    Supports:
    - Gzip compressed files (.gz)
    - Zip archives (.zip) with member selection
    - Tar archives (.tar, .tar.gz, .tgz, .tar.bz2) with member selection
    """

    @staticmethod
    def is_compressed(path: Path) -> bool:
        """Check if a file is compressed."""
        suffix = path.suffix.lower()
        return suffix in {".gz", ".gzip", ".zip", ".bz2", ".xz", ".tar", ".tgz"}

    @staticmethod
    def is_archive(path: Path) -> bool:
        """Check if a file is an archive (contains multiple files)."""
        suffix = path.suffix.lower()
        name_lower = path.name.lower()
        return suffix in {".zip", ".tar"} or name_lower.endswith((".tar.gz", ".tgz", ".tar.bz2"))

    @staticmethod
    def decompress_gzip(path: Path, encoding: str = "utf-8") -> str:
        """Decompress a gzip file and return content as string.

        Args:
            path: Path to the gzip file.
            encoding: Text encoding to use.

        Returns:
            Decompressed file content as string.
        """
        with gzip.open(path, "rt", encoding=encoding) as f:
            return f.read()

    @staticmethod
    def decompress_gzip_bytes(path: Path) -> bytes:
        """Decompress a gzip file and return content as bytes.

        Args:
            path: Path to the gzip file.

        Returns:
            Decompressed file content as bytes.
        """
        with gzip.open(path, "rb") as f:
            return f.read()

    @staticmethod
    def list_zip_members(path: Path) -> List[str]:
        """List members in a zip archive.

        Args:
            path: Path to the zip file.

        Returns:
            List of member names in the archive.
        """
        with zipfile.ZipFile(path, "r") as z:
            return z.namelist()

    @staticmethod
    def extract_from_zip(
        path: Path,
        member: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> str:
        """Extract a file from a zip archive.

        Args:
            path: Path to the zip file.
            member: Name of the member to extract. If None, auto-detect.
            encoding: Text encoding to use.

        Returns:
            Content of the extracted file as string.

        Raises:
            FileLoadError: If no suitable member is found.
        """
        with zipfile.ZipFile(path, "r") as z:
            members = z.namelist()

            if member is not None:
                if member not in members:
                    raise FileLoadError(
                        f"Member '{member}' not found in {path}. "
                        f"Available: {members}"
                    )
                return z.read(member).decode(encoding)

            # Auto-detect: find first CSV, then any file
            csv_files = [m for m in members if m.lower().endswith(".csv")]
            if csv_files:
                if len(csv_files) > 1:
                    import warnings
                    warnings.warn(
                        f"Multiple CSV files in {path}. Using {csv_files[0]}. "
                        f"Specify 'member' parameter to choose a specific file.",
                        UserWarning
                    )
                return z.read(csv_files[0]).decode(encoding)

            # Fall back to first non-directory entry
            data_files = [m for m in members if not m.endswith("/")]
            if not data_files:
                raise FileLoadError(f"No data files found in {path}")

            return z.read(data_files[0]).decode(encoding)

    @staticmethod
    def extract_bytes_from_zip(
        path: Path,
        member: Optional[str] = None,
    ) -> bytes:
        """Extract a file from a zip archive as bytes.

        Args:
            path: Path to the zip file.
            member: Name of the member to extract. If None, auto-detect.

        Returns:
            Content of the extracted file as bytes.
        """
        with zipfile.ZipFile(path, "r") as z:
            members = z.namelist()

            if member is not None:
                if member not in members:
                    raise FileLoadError(
                        f"Member '{member}' not found in {path}. "
                        f"Available: {members}"
                    )
                return z.read(member)

            # Auto-detect: find first matching file
            data_files = [m for m in members if not m.endswith("/")]
            if not data_files:
                raise FileLoadError(f"No data files found in {path}")

            return z.read(data_files[0])

    @staticmethod
    def list_tar_members(path: Path) -> List[str]:
        """List members in a tar archive.

        Args:
            path: Path to the tar file.

        Returns:
            List of member names in the archive.
        """
        mode = ArchiveHandler._get_tar_mode(path)
        with tarfile.open(path, mode) as t:
            return t.getnames()

    @staticmethod
    def extract_from_tar(
        path: Path,
        member: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> str:
        """Extract a file from a tar archive.

        Args:
            path: Path to the tar file.
            member: Name of the member to extract. If None, auto-detect.
            encoding: Text encoding to use.

        Returns:
            Content of the extracted file as string.

        Raises:
            FileLoadError: If no suitable member is found.
        """
        mode = ArchiveHandler._get_tar_mode(path)
        with tarfile.open(path, mode) as t:
            members = t.getnames()

            if member is not None:
                if member not in members:
                    raise FileLoadError(
                        f"Member '{member}' not found in {path}. "
                        f"Available: {members}"
                    )
                f = t.extractfile(member)
                if f is None:
                    raise FileLoadError(f"Cannot extract '{member}' from {path}")
                return f.read().decode(encoding)

            # Auto-detect: find first CSV, then any file
            csv_files = [m for m in members if m.lower().endswith(".csv")]
            if csv_files:
                if len(csv_files) > 1:
                    import warnings
                    warnings.warn(
                        f"Multiple CSV files in {path}. Using {csv_files[0]}. "
                        f"Specify 'member' parameter to choose a specific file.",
                        UserWarning
                    )
                f = t.extractfile(csv_files[0])
                if f is None:
                    raise FileLoadError(f"Cannot extract '{csv_files[0]}' from {path}")
                return f.read().decode(encoding)

            # Fall back to first regular file
            for name in members:
                info = t.getmember(name)
                if info.isfile():
                    f = t.extractfile(name)
                    if f is not None:
                        return f.read().decode(encoding)

            raise FileLoadError(f"No extractable files found in {path}")

    @staticmethod
    def extract_bytes_from_tar(
        path: Path,
        member: Optional[str] = None,
    ) -> bytes:
        """Extract a file from a tar archive as bytes.

        Args:
            path: Path to the tar file.
            member: Name of the member to extract. If None, auto-detect.

        Returns:
            Content of the extracted file as bytes.
        """
        mode = ArchiveHandler._get_tar_mode(path)
        with tarfile.open(path, mode) as t:
            members = t.getnames()

            if member is not None:
                if member not in members:
                    raise FileLoadError(
                        f"Member '{member}' not found in {path}. "
                        f"Available: {members}"
                    )
                f = t.extractfile(member)
                if f is None:
                    raise FileLoadError(f"Cannot extract '{member}' from {path}")
                return f.read()

            # Fall back to first regular file
            for name in members:
                info = t.getmember(name)
                if info.isfile():
                    f = t.extractfile(name)
                    if f is not None:
                        return f.read()

            raise FileLoadError(f"No extractable files found in {path}")

    @staticmethod
    def _get_tar_mode(path: Path) -> str:
        """Determine the correct mode for opening a tar file.

        Args:
            path: Path to the tar file.

        Returns:
            Mode string for tarfile.open().
        """
        name_lower = path.name.lower()
        if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
            return "r:gz"
        elif name_lower.endswith(".tar.bz2"):
            return "r:bz2"
        elif name_lower.endswith(".tar.xz"):
            return "r:xz"
        else:
            return "r"


class LoaderRegistry:
    """Registry for file loaders.

    The registry maintains a list of available loaders and provides
    methods for finding the appropriate loader for a given file.

    Example:
        >>> registry = LoaderRegistry()
        >>> registry.register(CSVLoader)
        >>> registry.register(ParquetLoader)
        >>> loader = registry.get_loader(Path("data.csv"))
        >>> result = loader.load(Path("data.csv"))
    """

    _instance: Optional["LoaderRegistry"] = None
    _loaders: List[Type[FileLoader]]

    def __new__(cls) -> "LoaderRegistry":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaders = []
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LoaderRegistry":
        """Get the singleton registry instance."""
        return cls()

    def register(self, loader_class: Type[FileLoader]) -> None:
        """Register a file loader.

        Args:
            loader_class: The loader class to register.
        """
        if loader_class not in self._loaders:
            self._loaders.append(loader_class)
            # Sort by priority (lower = higher priority)
            self._loaders.sort(key=lambda x: x.priority)

    def unregister(self, loader_class: Type[FileLoader]) -> None:
        """Unregister a file loader.

        Args:
            loader_class: The loader class to unregister.
        """
        if loader_class in self._loaders:
            self._loaders.remove(loader_class)

    def get_loader(self, path: Union[str, Path]) -> FileLoader:
        """Get the appropriate loader for a file.

        Args:
            path: Path to the file to load.

        Returns:
            An instance of the appropriate loader.

        Raises:
            FormatNotSupportedError: If no loader supports the file format.
        """
        path = Path(path)

        for loader_class in self._loaders:
            if loader_class.supports(path):
                return loader_class()

        raise FormatNotSupportedError(
            f"No loader found for file: {path}. "
            f"Supported formats: {self.get_supported_extensions()}"
        )

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions.

        Returns:
            List of supported extensions across all registered loaders.
        """
        extensions = set()
        for loader in self._loaders:
            extensions.update(loader.supported_extensions)
        return sorted(extensions)

    def get_registered_loaders(self) -> List[Type[FileLoader]]:
        """Get all registered loader classes.

        Returns:
            List of registered loader classes.
        """
        return list(self._loaders)

    def clear(self) -> None:
        """Clear all registered loaders (mainly for testing)."""
        self._loaders.clear()

    def load(
        self,
        path: Union[str, Path],
        **params: Any,
    ) -> LoaderResult:
        """Load a file using the appropriate loader.

        This is a convenience method that finds the right loader and loads the file.

        Args:
            path: Path to the file to load.
            **params: Loading parameters to pass to the loader.

        Returns:
            LoaderResult containing the loaded data.

        Raises:
            FormatNotSupportedError: If no loader supports the file format.
            FileLoadError: If the file cannot be loaded.
        """
        loader = self.get_loader(path)
        return loader.load(Path(path), **params)


def register_loader(cls: Type[FileLoader]) -> Type[FileLoader]:
    """Decorator to register a loader with the global registry.

    Example:
        >>> @register_loader
        ... class MyLoader(FileLoader):
        ...     ...
    """
    LoaderRegistry.get_instance().register(cls)
    return cls
