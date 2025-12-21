"""
Archive file loader implementation.

This module provides the ArchiveLoader class for loading data from archive files,
including tar (.tar, .tar.gz, .tgz, .tar.bz2) and enhanced zip support.

The ArchiveLoader acts as a wrapper that extracts files from archives and
delegates to the appropriate format-specific loader.
"""

import tarfile
import zipfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import pandas as pd

from .base import (
    ArchiveHandler,
    FileLoadError,
    FileLoader,
    LoaderRegistry,
    LoaderResult,
    register_loader,
)


@register_loader
class TarLoader(FileLoader):
    """Loader for tar archive files.

    Supports:
    - Plain tar files (.tar)
    - Gzip-compressed tar files (.tar.gz, .tgz)
    - Bzip2-compressed tar files (.tar.bz2)
    - XZ-compressed tar files (.tar.xz)

    Parameters:
        member: Name of the member file to extract. If None, auto-detects
            the first suitable file (prefers CSV).
        encoding: Text encoding for the extracted file (default: 'utf-8').
        inner_loader_params: Parameters to pass to the inner file loader.

    Example:
        >>> loader = TarLoader()
        >>> result = loader.load(
        ...     Path("data.tar.gz"),
        ...     member="data/train.csv",
        ... )
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = (".tar",)
    name: ClassVar[str] = "Tar Archive Loader"
    priority: ClassVar[int] = 60  # Lower priority - use specific loaders first

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file."""
        name_lower = path.name.lower()

        # Check common tar extensions
        if path.suffix.lower() == ".tar":
            return True
        if name_lower.endswith(".tar.gz"):
            return True
        if name_lower.endswith(".tgz"):
            return True
        if name_lower.endswith(".tar.bz2"):
            return True
        if name_lower.endswith(".tar.xz"):
            return True

        return False

    def load(
        self,
        path: Path,
        member: Optional[str] = None,
        encoding: str = "utf-8",
        header_unit: str = "cm-1",
        data_type: str = "x",
        **params: Any,
    ) -> LoaderResult:
        """Load data from a tar archive.

        Args:
            path: Path to the tar archive.
            member: Name of the member to extract. If None, auto-detects.
            encoding: Text encoding for extracted files.
            header_unit: Unit type for headers.
            data_type: Type of data ('x', 'y', or 'metadata').
            **params: Additional parameters for the inner loader.

        Returns:
            LoaderResult with the loaded data.
        """
        report: Dict[str, Any] = {
            "file_path": str(path),
            "format": "tar",
            "compression": self._detect_compression(path),
            "member_requested": member,
            "member_used": None,
            "members_available": None,
            "inner_format": None,
            "initial_shape": None,
            "final_shape": None,
            "na_handling": {
                "strategy": "remove",
                "na_detected": False,
                "nb_removed_rows": 0,
                "removed_rows_indices": [],
            },
            "warnings": [],
            "error": None,
        }

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Open tar file and list members
            mode = ArchiveHandler._get_tar_mode(path)

            try:
                with tarfile.open(path, mode) as t:
                    all_members = t.getnames()
                    file_members = [
                        m for m in all_members
                        if not m.endswith("/") and t.getmember(m).isfile()
                    ]
            except Exception as e:
                report["error"] = f"Failed to open tar archive: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            report["members_available"] = file_members

            if not file_members:
                report["error"] = "No files found in tar archive."
                return LoaderResult(report=report, header_unit=header_unit)

            # Select member to extract
            selected_member = self._select_member(member, file_members, report)
            if selected_member is None:
                return LoaderResult(report=report, header_unit=header_unit)

            report["member_used"] = selected_member

            # Determine inner format
            inner_ext = Path(selected_member).suffix.lower()
            report["inner_format"] = inner_ext

            # Extract content
            try:
                content = ArchiveHandler.extract_from_tar(path, selected_member, encoding)
            except Exception as e:
                report["error"] = f"Failed to extract '{selected_member}': {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            # For CSV files, use the CSV loader logic
            if inner_ext == ".csv":
                from .csv_loader_new import CSVLoader

                # Create a temporary path-like for the loader
                import io
                import tempfile
                import os

                # Write to temp file and load (simplest approach)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    delete=False,
                    encoding=encoding,
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    csv_loader = CSVLoader()
                    result = csv_loader.load(
                        Path(tmp_path),
                        header_unit=header_unit,
                        data_type=data_type,
                        **params,
                    )

                    # Update report with inner loader info, preserving archive info
                    result.report["outer_archive"] = str(path)
                    result.report["member_extracted"] = selected_member
                    result.report["members_available"] = file_members
                    report.update({
                        "initial_shape": result.report.get("initial_shape"),
                        "final_shape": result.report.get("final_shape"),
                        "na_handling": result.report.get("na_handling", report["na_handling"]),
                    })

                    return result

                finally:
                    os.unlink(tmp_path)

            else:
                # For non-CSV, try generic loading
                report["warnings"].append(
                    f"Non-CSV file extracted: {inner_ext}. "
                    f"Basic text loading applied."
                )

                # Try to parse as CSV anyway (many formats are CSV-like)
                from .csv_loader_new import CSVLoader
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=inner_ext,
                    delete=False,
                    encoding=encoding,
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    csv_loader = CSVLoader()
                    result = csv_loader.load(
                        Path(tmp_path),
                        header_unit=header_unit,
                        data_type=data_type,
                        **params,
                    )
                    # Preserve archive info in report
                    result.report["outer_archive"] = str(path)
                    result.report["member_extracted"] = selected_member
                    result.report["members_available"] = file_members
                    return result
                finally:
                    os.unlink(tmp_path)

        except FileNotFoundError as e:
            report["error"] = str(e)
            return LoaderResult(report=report, header_unit=header_unit)
        except Exception as e:
            import traceback
            report["error"] = f"Error loading tar archive: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)

    def _detect_compression(self, path: Path) -> str:
        """Detect compression type from file name."""
        name_lower = path.name.lower()
        if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
            return "gzip"
        elif name_lower.endswith(".tar.bz2"):
            return "bzip2"
        elif name_lower.endswith(".tar.xz"):
            return "xz"
        else:
            return "none"

    def _select_member(
        self,
        requested: Optional[str],
        available: List[str],
        report: Dict[str, Any],
    ) -> Optional[str]:
        """Select which member to extract."""
        if requested is not None:
            if requested in available:
                return requested
            report["error"] = (
                f"Member '{requested}' not found in archive. "
                f"Available: {available}"
            )
            return None

        # Auto-select: prefer CSV, then other common formats
        preferred_extensions = [".csv", ".tsv", ".txt", ".dat"]

        for ext in preferred_extensions:
            matches = [m for m in available if m.lower().endswith(ext)]
            if matches:
                if len(matches) > 1:
                    report["warnings"].append(
                        f"Multiple {ext} files found. Using '{matches[0]}'. "
                        f"Specify 'member' to choose a specific file."
                    )
                return matches[0]

        # Fall back to first file
        return available[0]


@register_loader
class EnhancedZipLoader(FileLoader):
    """Enhanced loader for zip archive files.

    This loader provides additional features over the basic zip support
    in the CSV loader, including:
    - Member listing and selection
    - Support for non-CSV files in archives
    - Binary file extraction (for NumPy, Parquet, etc.)

    Parameters:
        member: Name of the member file to extract.
        password: Password for encrypted archives.
        encoding: Text encoding for text files.

    Example:
        >>> loader = EnhancedZipLoader()
        >>> result = loader.load(
        ...     Path("data.zip"),
        ...     member="train/features.csv",
        ... )
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = (".zip",)
    name: ClassVar[str] = "Enhanced Zip Loader"
    priority: ClassVar[int] = 65  # Lower priority than specific loaders

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file."""
        suffix = path.suffix.lower()

        # Only handle .zip files that aren't already handled by format-specific loaders
        if suffix == ".zip":
            # Check if it's a format-specific zip (like .csv.zip)
            name_lower = path.name.lower()
            if name_lower.endswith(".csv.zip"):
                return False  # Let CSVLoader handle this
            if name_lower.endswith(".npy.zip") or name_lower.endswith(".npz.zip"):
                return False  # Let NumpyLoader handle this
            return True

        return False

    def load(
        self,
        path: Path,
        member: Optional[str] = None,
        password: Optional[str] = None,
        encoding: str = "utf-8",
        header_unit: str = "cm-1",
        data_type: str = "x",
        **params: Any,
    ) -> LoaderResult:
        """Load data from a zip archive.

        Args:
            path: Path to the zip archive.
            member: Name of the member to extract.
            password: Password for encrypted archives.
            encoding: Text encoding for text files.
            header_unit: Unit type for headers.
            data_type: Type of data.
            **params: Additional parameters for the inner loader.

        Returns:
            LoaderResult with the loaded data.
        """
        report: Dict[str, Any] = {
            "file_path": str(path),
            "format": "zip",
            "member_requested": member,
            "member_used": None,
            "members_available": None,
            "inner_format": None,
            "initial_shape": None,
            "final_shape": None,
            "na_handling": {
                "strategy": "remove",
                "na_detected": False,
                "nb_removed_rows": 0,
                "removed_rows_indices": [],
            },
            "warnings": [],
            "error": None,
        }

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Open zip and list members
            try:
                with zipfile.ZipFile(path, "r") as z:
                    all_members = z.namelist()
                    file_members = [m for m in all_members if not m.endswith("/")]
            except zipfile.BadZipFile as e:
                report["error"] = f"Invalid zip file: {e}"
                return LoaderResult(report=report, header_unit=header_unit)
            except Exception as e:
                report["error"] = f"Failed to open zip archive: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            report["members_available"] = file_members

            if not file_members:
                report["error"] = "No files found in zip archive."
                return LoaderResult(report=report, header_unit=header_unit)

            # Select member
            selected_member = self._select_member(member, file_members, report)
            if selected_member is None:
                return LoaderResult(report=report, header_unit=header_unit)

            report["member_used"] = selected_member

            # Determine inner format
            inner_ext = Path(selected_member).suffix.lower()
            report["inner_format"] = inner_ext

            # Extract and load based on format
            pwd = password.encode() if password else None

            try:
                with zipfile.ZipFile(path, "r") as z:
                    content_bytes = z.read(selected_member, pwd=pwd)
            except RuntimeError as e:
                if "password" in str(e).lower():
                    report["error"] = "Archive is encrypted. Provide 'password' parameter."
                else:
                    report["error"] = f"Failed to extract '{selected_member}': {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            # Handle based on inner format
            return self._load_inner_content(
                content_bytes,
                inner_ext,
                encoding,
                header_unit,
                data_type,
                report,
                **params,
            )

        except FileNotFoundError as e:
            report["error"] = str(e)
            return LoaderResult(report=report, header_unit=header_unit)
        except Exception as e:
            import traceback
            report["error"] = f"Error loading zip archive: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)

    def _select_member(
        self,
        requested: Optional[str],
        available: List[str],
        report: Dict[str, Any],
    ) -> Optional[str]:
        """Select which member to extract."""
        if requested is not None:
            if requested in available:
                return requested
            report["error"] = (
                f"Member '{requested}' not found in archive. "
                f"Available: {available}"
            )
            return None

        # Auto-select priority
        preferred_extensions = [".csv", ".parquet", ".pq", ".npy", ".npz", ".xlsx", ".mat"]

        for ext in preferred_extensions:
            matches = [m for m in available if m.lower().endswith(ext)]
            if matches:
                if len(matches) > 1:
                    report["warnings"].append(
                        f"Multiple {ext} files found. Using '{matches[0]}'. "
                        f"Specify 'member' to choose a specific file."
                    )
                return matches[0]

        # Fall back to first file
        return available[0]

    def _load_inner_content(
        self,
        content_bytes: bytes,
        inner_ext: str,
        encoding: str,
        header_unit: str,
        data_type: str,
        report: Dict[str, Any],
        **params: Any,
    ) -> LoaderResult:
        """Load content based on inner file format."""
        import tempfile
        import os

        # Write to temp file for loader
        with tempfile.NamedTemporaryFile(
            suffix=inner_ext,
            delete=False,
        ) as tmp:
            tmp.write(content_bytes)
            tmp_path = Path(tmp.name)

        try:
            # Try to get the appropriate loader from registry
            registry = LoaderRegistry.get_instance()

            try:
                loader = registry.get_loader(tmp_path)
            except Exception:
                # Fall back to CSV loader
                from .csv_loader_new import CSVLoader
                loader = CSVLoader()

            result = loader.load(
                tmp_path,
                header_unit=header_unit,
                data_type=data_type,
                encoding=encoding,
                **params,
            )

            # Preserve archive info in result report
            result.report["members_available"] = report.get("members_available", [])
            result.report["member_used"] = report.get("member_used")

            # Update report
            report.update({
                "initial_shape": result.report.get("initial_shape"),
                "final_shape": result.report.get("final_shape"),
                "na_handling": result.report.get("na_handling", report["na_handling"]),
            })

            return result

        finally:
            os.unlink(tmp_path)


def list_archive_members(path) -> List[str]:
    """List members in an archive file.

    Args:
        path: Path to the archive.

    Returns:
        List of member names.

    Raises:
        FileLoadError: If the archive cannot be read.
    """
    path = Path(path)

    if path.suffix.lower() == ".zip" or path.name.lower().endswith(".zip"):
        return ArchiveHandler.list_zip_members(path)

    if TarLoader.supports(path):
        return ArchiveHandler.list_tar_members(path)

    raise FileLoadError(f"Unknown archive format: {path}")
