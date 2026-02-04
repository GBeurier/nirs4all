"""Tests for chain_replay module.

Verifies that the ``replay_chain`` function correctly delegates to
WorkspaceStore.replay_chain() and returns predicted values.
"""

from unittest.mock import MagicMock

import numpy as np

from nirs4all.pipeline.storage.chain_replay import replay_chain


class TestReplayChain:
    """Tests for the replay_chain helper function."""

    def test_delegates_to_store(self):
        """replay_chain should call store.replay_chain with correct args."""
        store = MagicMock()
        expected = np.array([1.0, 2.0, 3.0])
        store.replay_chain.return_value = expected

        X = np.random.randn(3, 10)
        result = replay_chain(store, chain_id="abc123", X=X)

        store.replay_chain.assert_called_once_with(
            chain_id="abc123", X=X, wavelengths=None
        )
        np.testing.assert_array_equal(result, expected)

    def test_passes_wavelengths(self):
        """replay_chain should forward wavelengths to the store."""
        store = MagicMock()
        store.replay_chain.return_value = np.array([0.5])

        X = np.random.randn(1, 5)
        wl = np.linspace(400, 2500, 5)

        replay_chain(store, chain_id="def456", X=X, wavelengths=wl)

        call_kwargs = store.replay_chain.call_args
        np.testing.assert_array_equal(call_kwargs.kwargs["wavelengths"], wl)

    def test_returns_numpy_array(self):
        """replay_chain should return a numpy array."""
        store = MagicMock()
        store.replay_chain.return_value = np.zeros(5)

        result = replay_chain(store, chain_id="xyz", X=np.zeros((5, 10)))

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
