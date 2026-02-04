"""Chain replay -- predict from stored chains via WorkspaceStore.

This module provides a standalone ``replay_chain`` function that loads a chain
from the DuckDB-backed WorkspaceStore and applies it to new feature data.

The function is the primary in-workspace prediction path: it replays
preprocessing transformers in order and averages fold-model predictions.

For out-of-workspace prediction, export to ``.n4a`` first via
``WorkspaceStore.export_chain``.

Example:
    >>> from nirs4all.pipeline.storage import WorkspaceStore
    >>> from nirs4all.pipeline.storage.chain_replay import replay_chain
    >>>
    >>> store = WorkspaceStore(workspace_path)
    >>> y_pred = replay_chain(store, chain_id="abc123", X=X_new)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def replay_chain(
    store: WorkspaceStore,
    chain_id: str,
    X: np.ndarray,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Replay a stored chain on new data to produce predictions.

    Delegates to ``WorkspaceStore.replay_chain`` which loads each step's
    artifact, applies the transformation in order, and for the model step
    loads all fold models and returns the averaged prediction.

    Args:
        store: Open WorkspaceStore instance.
        chain_id: Chain to replay.
        X: Input feature matrix (``n_samples x n_features``).
        wavelengths: Optional wavelength array for wavelength-aware
            operators.

    Returns:
        Predicted values as a 1-D ``numpy.ndarray`` of shape
        ``(n_samples,)``.

    Raises:
        KeyError: If the chain does not exist.
        RuntimeError: If the chain has no model step or fold artifacts.
    """
    return store.replay_chain(chain_id=chain_id, X=X, wavelengths=wavelengths)
