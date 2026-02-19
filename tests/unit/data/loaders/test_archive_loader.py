"""
Unit tests for archive loaders (Tar and enhanced Zip).

Tests loading data from .tar, .tar.gz, .tgz, and .zip archives.
"""

import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.loaders.archive_loader import (
    EnhancedZipLoader,
    TarLoader,
    list_archive_members,
)


class TestTarLoaderSupports:
    """Tests for TarLoader.supports() method."""

    def test_supports_tar(self):
        """Test that TarLoader supports .tar files."""
        assert TarLoader.supports(Path("data.tar"))

    def test_supports_tar_gz(self):
        """Test that TarLoader supports .tar.gz files."""
        assert TarLoader.supports(Path("data.tar.gz"))

    def test_supports_tgz(self):
        """Test that TarLoader supports .tgz files."""
        assert TarLoader.supports(Path("data.tgz"))

    def test_supports_tar_bz2(self):
        """Test that TarLoader supports .tar.bz2 files."""
        assert TarLoader.supports(Path("data.tar.bz2"))

    def test_not_supports_other(self):
        """Test that TarLoader doesn't support other formats."""
        assert not TarLoader.supports(Path("data.zip"))
        assert not TarLoader.supports(Path("data.gz"))  # Plain gz without tar
        assert not TarLoader.supports(Path("data.csv"))

class TestTarLoaderLoad:
    """Tests for TarLoader.load() method."""

    @pytest.fixture
    def tar_with_csv(self):
        """Create a tar archive with a CSV file."""
        csv_content = "a;b;c\n1.0;2.0;3.0\n4.0;5.0;6.0"

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
            with tarfile.open(f.name, "w") as t:
                import io
                csv_bytes = csv_content.encode("utf-8")
                info = tarfile.TarInfo(name="data.csv")
                info.size = len(csv_bytes)
                t.addfile(info, io.BytesIO(csv_bytes))
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def tar_gz_with_csv(self):
        """Create a tar.gz archive with a CSV file."""
        csv_content = "a;b;c\n1.0;2.0;3.0\n4.0;5.0;6.0"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            with tarfile.open(f.name, "w:gz") as t:
                import io
                csv_bytes = csv_content.encode("utf-8")
                info = tarfile.TarInfo(name="data.csv")
                info.size = len(csv_bytes)
                t.addfile(info, io.BytesIO(csv_bytes))
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def tar_with_multiple_files(self):
        """Create a tar archive with multiple files."""
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
            with tarfile.open(f.name, "w") as t:
                import io
                for name, content in [
                    ("file1.csv", "a;b\n1;2"),
                    ("file2.csv", "c;d\n3;4"),
                    ("readme.txt", "This is a readme"),
                ]:
                    data = content.encode("utf-8")
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_tar(self, tar_with_csv):
        """Test loading a tar archive."""
        loader = TarLoader()
        result = loader.load(tar_with_csv)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (2, 3)

    def test_load_tar_gz(self, tar_gz_with_csv):
        """Test loading a tar.gz archive."""
        loader = TarLoader()
        result = loader.load(tar_gz_with_csv)

        assert result.success
        assert result.data.shape == (2, 3)

    def test_load_specific_member(self, tar_with_multiple_files):
        """Test loading a specific member from archive."""
        loader = TarLoader()
        result = loader.load(tar_with_multiple_files, member="file2.csv")

        assert result.success
        assert list(result.data.columns) == ["c", "d"]

    def test_load_invalid_member(self, tar_with_csv):
        """Test loading with invalid member name."""
        loader = TarLoader()
        result = loader.load(tar_with_csv, member="nonexistent.csv")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = TarLoader()
        result = loader.load(Path("/nonexistent/file.tar"))

        assert not result.success
        assert "not found" in result.error.lower()

    def test_report_contains_members(self, tar_with_multiple_files):
        """Test that report contains member list."""
        loader = TarLoader()
        result = loader.load(tar_with_multiple_files)

        assert "file1.csv" in result.report.get("members_available", [])
        assert "file2.csv" in result.report.get("members_available", [])

class TestEnhancedZipLoaderSupports:
    """Tests for EnhancedZipLoader.supports() method."""

    def test_supports_zip(self):
        """Test that EnhancedZipLoader supports .zip files."""
        assert EnhancedZipLoader.supports(Path("data.zip"))

    def test_not_supports_csv_zip(self):
        """Test that csv.zip is handled by CSVLoader instead."""
        assert not EnhancedZipLoader.supports(Path("data.csv.zip"))

    def test_not_supports_other(self):
        """Test that EnhancedZipLoader doesn't support other formats."""
        assert not EnhancedZipLoader.supports(Path("data.tar"))
        assert not EnhancedZipLoader.supports(Path("data.csv"))

class TestEnhancedZipLoaderLoad:
    """Tests for EnhancedZipLoader.load() method."""

    @pytest.fixture
    def zip_with_csv(self):
        """Create a zip archive with a CSV file."""
        csv_content = "a;b;c\n1.0;2.0;3.0\n4.0;5.0;6.0"

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            with zipfile.ZipFile(f.name, "w") as z:
                z.writestr("data.csv", csv_content)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def zip_with_multiple_files(self):
        """Create a zip archive with multiple files."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            with zipfile.ZipFile(f.name, "w") as z:
                z.writestr("train.csv", "a;b\n1;2")
                z.writestr("test.csv", "c;d\n3;4")
                z.writestr("readme.txt", "This is a readme")
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_zip(self, zip_with_csv):
        """Test loading a zip archive."""
        loader = EnhancedZipLoader()
        result = loader.load(zip_with_csv)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (2, 3)

    def test_load_specific_member(self, zip_with_multiple_files):
        """Test loading a specific member from archive."""
        loader = EnhancedZipLoader()
        result = loader.load(zip_with_multiple_files, member="test.csv")

        assert result.success
        assert list(result.data.columns) == ["c", "d"]

    def test_load_invalid_member(self, zip_with_csv):
        """Test loading with invalid member name."""
        loader = EnhancedZipLoader()
        result = loader.load(zip_with_csv, member="nonexistent.csv")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = EnhancedZipLoader()
        result = loader.load(Path("/nonexistent/file.zip"))

        assert not result.success
        assert "not found" in result.error.lower()

class TestListArchiveMembers:
    """Tests for list_archive_members function."""

    @pytest.fixture
    def zip_archive(self):
        """Create a zip archive."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            with zipfile.ZipFile(f.name, "w") as z:
                z.writestr("file1.csv", "data")
                z.writestr("file2.csv", "data")
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def tar_archive(self):
        """Create a tar archive."""
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
            with tarfile.open(f.name, "w") as t:
                import io
                for name in ["file1.csv", "file2.csv"]:
                    data = b"data"
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
            yield Path(f.name)
        Path(f.name).unlink()

    def test_list_zip_members(self, zip_archive):
        """Test listing zip archive members."""
        members = list_archive_members(zip_archive)

        assert "file1.csv" in members
        assert "file2.csv" in members

    def test_list_tar_members(self, tar_archive):
        """Test listing tar archive members."""
        members = list_archive_members(tar_archive)

        assert "file1.csv" in members
        assert "file2.csv" in members
