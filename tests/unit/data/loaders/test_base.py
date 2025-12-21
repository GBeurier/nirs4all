"""
Unit tests for the base loader classes and registry.

Tests the FileLoader base class, LoaderRegistry, LoaderResult,
and ArchiveHandler utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from nirs4all.data.loaders.base import (
    ArchiveHandler,
    FileLoadError,
    FileLoader,
    FormatNotSupportedError,
    LoaderError,
    LoaderRegistry,
    LoaderResult,
    register_loader,
)


class TestLoaderResult:
    """Tests for LoaderResult class."""

    def test_success_when_data_present(self):
        """Test that success is True when data is present and no error."""
        result = LoaderResult(
            data=pd.DataFrame({"a": [1, 2, 3]}),
            report={},
        )
        assert result.success is True
        assert result.error is None

    def test_failure_when_data_none(self):
        """Test that success is False when data is None."""
        result = LoaderResult(data=None, report={})
        assert result.success is False

    def test_failure_when_error_in_report(self):
        """Test that success is False when error is in report."""
        result = LoaderResult(
            data=pd.DataFrame({"a": [1, 2, 3]}),
            report={"error": "Something went wrong"},
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_default_values(self):
        """Test default values for LoaderResult."""
        result = LoaderResult()
        assert result.data is None
        assert result.report == {}
        assert result.na_mask is None
        assert result.headers == []
        assert result.header_unit == "cm-1"


class TestFileLoaderBase:
    """Tests for FileLoader abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that FileLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FileLoader()

    def test_get_base_path(self):
        """Test get_base_path removes compression extensions."""
        assert FileLoader.get_base_path(Path("data.csv")) == Path("data.csv")
        assert FileLoader.get_base_path(Path("data.csv.gz")) == Path("data.csv")
        assert FileLoader.get_base_path(Path("data.csv.zip")) == Path("data.csv")
        assert FileLoader.get_base_path(Path("data.npy.gz")) == Path("data.npy")


class TestLoaderRegistry:
    """Tests for LoaderRegistry class."""

    @pytest.fixture
    def clean_registry(self):
        """Provide a clean registry for each test."""
        registry = LoaderRegistry.get_instance()
        original_loaders = registry._loaders.copy()
        registry.clear()
        yield registry
        # Restore original loaders
        registry._loaders = original_loaders

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        r1 = LoaderRegistry.get_instance()
        r2 = LoaderRegistry.get_instance()
        assert r1 is r2

    def test_register_and_unregister(self, clean_registry):
        """Test registering and unregistering loaders."""
        class MockLoader(FileLoader):
            supported_extensions = (".mock",)
            name = "Mock Loader"

            @classmethod
            def supports(cls, path):
                return path.suffix == ".mock"

            def load(self, path, **params):
                return LoaderResult()

        clean_registry.register(MockLoader)
        assert MockLoader in clean_registry.get_registered_loaders()

        clean_registry.unregister(MockLoader)
        assert MockLoader not in clean_registry.get_registered_loaders()

    def test_get_loader_not_found(self, clean_registry):
        """Test that FormatNotSupportedError is raised for unknown formats."""
        with pytest.raises(FormatNotSupportedError):
            clean_registry.get_loader(Path("data.unknown"))

    def test_get_supported_extensions(self, clean_registry):
        """Test getting supported extensions."""
        class MockLoader(FileLoader):
            supported_extensions = (".mock", ".mk")
            name = "Mock Loader"

            @classmethod
            def supports(cls, path):
                return path.suffix in cls.supported_extensions

            def load(self, path, **params):
                return LoaderResult()

        clean_registry.register(MockLoader)
        extensions = clean_registry.get_supported_extensions()
        assert ".mock" in extensions
        assert ".mk" in extensions

    def test_priority_ordering(self, clean_registry):
        """Test that loaders are ordered by priority."""
        class HighPriorityLoader(FileLoader):
            supported_extensions = (".test",)
            name = "High Priority"
            priority = 10

            @classmethod
            def supports(cls, path):
                return True

            def load(self, path, **params):
                return LoaderResult()

        class LowPriorityLoader(FileLoader):
            supported_extensions = (".test",)
            name = "Low Priority"
            priority = 100

            @classmethod
            def supports(cls, path):
                return True

            def load(self, path, **params):
                return LoaderResult()

        clean_registry.register(LowPriorityLoader)
        clean_registry.register(HighPriorityLoader)

        loaders = clean_registry.get_registered_loaders()
        high_idx = loaders.index(HighPriorityLoader)
        low_idx = loaders.index(LowPriorityLoader)
        assert high_idx < low_idx


class TestRegisterLoaderDecorator:
    """Tests for the @register_loader decorator."""

    def test_decorator_registers_class(self):
        """Test that decorator registers the class with the registry."""
        registry = LoaderRegistry.get_instance()
        original_count = len(registry.get_registered_loaders())

        @register_loader
        class DecoratedLoader(FileLoader):
            supported_extensions = (".decorated",)
            name = "Decorated Loader"

            @classmethod
            def supports(cls, path):
                return path.suffix == ".decorated"

            def load(self, path, **params):
                return LoaderResult()

        # Check it was registered
        assert DecoratedLoader in registry.get_registered_loaders()

        # Clean up
        registry.unregister(DecoratedLoader)


class TestArchiveHandler:
    """Tests for ArchiveHandler utility class."""

    def test_is_compressed(self):
        """Test is_compressed detection."""
        assert ArchiveHandler.is_compressed(Path("data.gz"))
        assert ArchiveHandler.is_compressed(Path("data.zip"))
        assert ArchiveHandler.is_compressed(Path("data.tar"))
        assert not ArchiveHandler.is_compressed(Path("data.csv"))

    def test_is_archive(self):
        """Test is_archive detection."""
        assert ArchiveHandler.is_archive(Path("data.zip"))
        assert ArchiveHandler.is_archive(Path("data.tar"))
        assert ArchiveHandler.is_archive(Path("data.tar.gz"))
        assert ArchiveHandler.is_archive(Path("data.tgz"))
        assert not ArchiveHandler.is_archive(Path("data.gz"))  # Single file compression

    def test_get_tar_mode(self):
        """Test tar mode detection."""
        assert ArchiveHandler._get_tar_mode(Path("data.tar")) == "r"
        assert ArchiveHandler._get_tar_mode(Path("data.tar.gz")) == "r:gz"
        assert ArchiveHandler._get_tar_mode(Path("data.tgz")) == "r:gz"
        assert ArchiveHandler._get_tar_mode(Path("data.tar.bz2")) == "r:bz2"
        assert ArchiveHandler._get_tar_mode(Path("data.tar.xz")) == "r:xz"
