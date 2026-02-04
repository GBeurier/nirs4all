"""Tests for store-based chain export via BundleGenerator.

Verifies that BundleGenerator.export_from_chain() correctly delegates
to WorkspaceStore.export_chain() for .n4a format export.
"""

from unittest.mock import MagicMock

import pytest

from nirs4all.pipeline.bundle.generator import BundleGenerator


class TestExportFromChain:
    """Tests for BundleGenerator.export_from_chain()."""

    def test_requires_store(self, tmp_path):
        """export_from_chain should raise if no store is set."""
        gen = BundleGenerator(workspace_path=tmp_path)
        with pytest.raises(RuntimeError, match="store"):
            gen.export_from_chain("abc123", tmp_path / "out.n4a")

    def test_delegates_to_store_for_n4a(self, tmp_path):
        """export_from_chain should call store.export_chain for .n4a format."""
        store = MagicMock()
        expected_path = tmp_path / "model.n4a"
        store.export_chain.return_value = expected_path

        gen = BundleGenerator(workspace_path=tmp_path, store=store)
        result = gen.export_from_chain("chain_abc", expected_path, fmt="n4a")

        store.export_chain.assert_called_once_with("chain_abc", expected_path, format="n4a")
        assert result == expected_path

    def test_default_format_is_n4a(self, tmp_path):
        """export_from_chain should default to n4a format."""
        store = MagicMock()
        store.export_chain.return_value = tmp_path / "out.n4a"

        gen = BundleGenerator(workspace_path=tmp_path, store=store)
        gen.export_from_chain("chain_xyz", tmp_path / "out.n4a")

        store.export_chain.assert_called_once()
