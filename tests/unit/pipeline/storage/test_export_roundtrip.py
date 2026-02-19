"""Round-trip tests for chain export and replay.

Verifies that the store-based export and chain replay paths work
together correctly, and that the API-level predict() function properly
handles chain_id-based predictions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.api.predict import _extract_X, _predict_from_chain, predict
from nirs4all.api.result import PredictResult


class TestExtractX:
    """Tests for the _extract_X helper."""

    def test_ndarray_passthrough(self):
        """numpy arrays should pass through unchanged."""
        X = np.random.randn(10, 5)
        result = _extract_X(X)
        np.testing.assert_array_equal(result, X)

    def test_tuple_extracts_first(self):
        """Tuples should extract the first element."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        result = _extract_X((X, y))
        np.testing.assert_array_equal(result, X)

    def test_dict_with_X_key(self):
        """Dicts with 'X' key should extract X."""
        X = np.random.randn(10, 5)
        result = _extract_X({"X": X, "metadata": {"name": "test"}})
        np.testing.assert_array_equal(result, X)

    def test_dict_without_X_key_raises(self):
        """Dicts without 'X' key should raise TypeError."""
        with pytest.raises(TypeError, match="'X' key"):
            _extract_X({"data": np.zeros(5)})

    def test_unsupported_type_raises(self):
        """Unsupported types should raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            _extract_X("not_supported_for_chain")

class TestPredictValidation:
    """Tests for the predict() function argument validation."""

    def test_requires_model_or_chain_id(self):
        """predict should raise if neither model nor chain_id is provided."""
        with pytest.raises(ValueError, match="Provide either"):
            predict(data=np.zeros((5, 10)))

    def test_rejects_both_model_and_chain_id(self):
        """predict should raise if both model and chain_id are provided."""
        with pytest.raises(ValueError, match="not both"):
            predict(model="test.n4a", chain_id="abc123", data=np.zeros((5, 10)))

    def test_requires_data(self):
        """predict should raise if data is not provided."""
        with pytest.raises(ValueError, match="data"):
            predict(chain_id="abc123")

class TestPredictFromChain:
    """Tests for the chain-based prediction path."""

    @patch("nirs4all.pipeline.storage.chain_replay.replay_chain")
    @patch("nirs4all.pipeline.storage.workspace_store.WorkspaceStore")
    def test_returns_predict_result(self, MockStore, mock_replay):
        """_predict_from_chain should return a PredictResult."""
        store_instance = MockStore.return_value
        mock_replay.return_value = np.array([1.0, 2.0])
        store_instance.get_chain.return_value = {"model_class": "PLSRegression"}

        X = np.random.randn(2, 10)
        result = _predict_from_chain(
            chain_id="abc123",
            data=X,
            workspace_path=Path("workspace"),
            session=None,
            verbose=0,
        )

        assert isinstance(result, PredictResult)
        np.testing.assert_array_equal(result.y_pred, np.array([1.0, 2.0]))
        assert result.metadata["chain_id"] == "abc123"
        mock_replay.assert_called_once()

    @patch("nirs4all.pipeline.storage.chain_replay.replay_chain")
    @patch("nirs4all.pipeline.storage.workspace_store.WorkspaceStore")
    def test_uses_session_workspace(self, MockStore, mock_replay):
        """_predict_from_chain should use session's workspace path when provided."""
        session = MagicMock()
        session.runner.workspace_path = Path("/session/workspace")

        store_instance = MockStore.return_value
        mock_replay.return_value = np.array([0.5])
        store_instance.get_chain.return_value = None

        X = np.random.randn(1, 5)
        _predict_from_chain(
            chain_id="test",
            data=X,
            workspace_path=None,
            session=session,
            verbose=0,
        )

        MockStore.assert_called_once_with(Path("/session/workspace"))
