"""Unit tests for ProcessingChain."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from nirs4all.data._targets.processing_chain import ProcessingChain


class TestProcessingChain:
    """Test suite for ProcessingChain class."""

    def test_add_processing(self):
        """Test adding processing steps to the chain."""
        chain = ProcessingChain()

        # Add root processing (no ancestor)
        chain.add_processing("raw", ancestor=None, transformer=None)
        assert chain.has_processing("raw")

        # Add child processing
        scaler1 = StandardScaler()
        chain.add_processing("scaled", ancestor="raw", transformer=scaler1)
        assert chain.has_processing("scaled")
        assert chain.get_transformer("scaled") is scaler1

        # Add grandchild processing
        scaler2 = StandardScaler()
        chain.add_processing("normalized", ancestor="scaled", transformer=scaler2)
        assert chain.has_processing("normalized")

    def test_add_processing_duplicate_name(self):
        """Test that duplicate processing names raise error."""
        chain = ProcessingChain()
        chain.add_processing("test", ancestor=None, transformer=None)

        with pytest.raises(ValueError, match="already exists"):
            chain.add_processing("test", ancestor=None, transformer=None)

    def test_add_processing_missing_ancestor(self):
        """Test that invalid ancestor raises error."""
        chain = ProcessingChain()

        scaler = StandardScaler()
        with pytest.raises(ValueError, match="does not exist"):
            chain.add_processing("scaled", ancestor="nonexistent", transformer=scaler)

    def test_get_ancestry_simple(self):
        """Test getting ancestry chain."""
        chain = ProcessingChain()
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("level1", ancestor="raw", transformer=scaler1)
        chain.add_processing("level2", ancestor="level1", transformer=scaler2)

        # Get ancestry for level2 (returns list of processing names)
        ancestry = chain.get_ancestry("level2")

        assert len(ancestry) == 3
        assert ancestry == ["raw", "level1", "level2"]

    def test_get_ancestry_caching(self):
        """Test that ancestry lookup uses caching."""
        chain = ProcessingChain()
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("level1", ancestor="raw", transformer=scaler1)
        chain.add_processing("level2", ancestor="level1", transformer=scaler2)

        # First call - should compute and cache
        ancestry1 = chain.get_ancestry("level2")

        # Second call - should use cache (returns copy so not same object)
        ancestry2 = chain.get_ancestry("level2")

        # Should have same content
        assert ancestry1 == ancestry2

    def test_get_path_forward(self):
        """Test getting forward transformation path."""
        chain = ProcessingChain()
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler1)
        chain.add_processing("normalized", ancestor="scaled", transformer=scaler2)

        # Get path from raw to normalized (returns path list and direction string)
        path, direction = chain.get_path("raw", "normalized")

        assert path is not None
        assert direction == "forward"
        assert "raw" in path
        assert "normalized" in path

    def test_get_path_inverse(self):
        """Test getting inverse transformation path."""
        chain = ProcessingChain()
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler1)
        chain.add_processing("normalized", ancestor="scaled", transformer=scaler2)

        # Get path from normalized to raw
        path, direction = chain.get_path("normalized", "raw")

        assert path is not None
        assert direction == "inverse"
        assert "raw" in path
        assert "normalized" in path

    def test_get_path_same_node(self):
        """Test get_path when source and target are the same."""
        chain = ProcessingChain()
        scaler = StandardScaler()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler)

        # Path from node to itself
        path, direction = chain.get_path("raw", "raw")

        assert path is not None
        assert direction == "identity"
        assert path == ["raw"]

    def test_get_path_no_path(self):
        """Test get_path when no path exists."""
        chain = ProcessingChain()
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        # Create two separate trees
        chain.add_processing("raw1", ancestor=None, transformer=None)
        chain.add_processing("scaled1", ancestor="raw1", transformer=scaler1)
        chain.add_processing("raw2", ancestor=None, transformer=None)
        chain.add_processing("scaled2", ancestor="raw2", transformer=scaler2)

        # No path between separate trees should raise ValueError
        with pytest.raises(ValueError, match="No common ancestor"):
            chain.get_path("scaled1", "scaled2")

    def test_has_processing(self):
        """Test has_processing method."""
        chain = ProcessingChain()
        scaler = StandardScaler()

        # Initially doesn't have processing
        assert not chain.has_processing("raw")

        # Add processing
        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler)

        # Now has both
        assert chain.has_processing("raw")
        assert chain.has_processing("scaled")

    def test_complex_branching(self):
        """Test complex processing chain with branches."""
        chain = ProcessingChain()

        # Create branching structure
        #       raw
        #      /   \
        #   branch1 branch2

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("branch1", ancestor="raw", transformer=scaler1)
        chain.add_processing("branch2", ancestor="raw", transformer=scaler2)

        # Test paths between branches
        path, direction = chain.get_path("branch1", "branch2")
        assert path is not None
        assert direction == "mixed"
        assert "raw" in path  # Common ancestor

    def test_num_processings(self):
        """Test num_processings property."""
        chain = ProcessingChain()

        assert chain.num_processings == 0

        chain.add_processing("raw", ancestor=None, transformer=None)
        assert chain.num_processings == 1

        scaler = StandardScaler()
        chain.add_processing("scaled", ancestor="raw", transformer=scaler)
        assert chain.num_processings == 2

    def test_processing_ids(self):
        """Test processing_ids property."""
        chain = ProcessingChain()

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=StandardScaler())

        ids = chain.processing_ids
        assert "raw" in ids
        assert "scaled" in ids
        assert len(ids) == 2
