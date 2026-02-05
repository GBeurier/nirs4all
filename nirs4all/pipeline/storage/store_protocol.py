"""Protocol definition for workspace storage backends.

Provides a structural subtyping contract (``typing.Protocol``) that any
storage backend must satisfy.  This enables testing with in-memory stubs
and, in the future, swapping DuckDB for another backend without changing
consumer code.

The protocol captures the *minimal* set of methods that all consumers
depend on.  ``WorkspaceStore`` itself implements a much richer interface
(export, replay, logging, cleanup); those additional methods are not part
of the protocol because backend-swappability is only required for the
core storage operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class WorkspaceStoreProtocol(Protocol):
    """Minimal protocol for workspace storage backends.

    A backend is any object that satisfies this structural interface.
    Use ``isinstance(obj, WorkspaceStoreProtocol)`` at runtime to verify
    compliance (enabled by ``@runtime_checkable``).

    The protocol covers:

    * **Run lifecycle** -- creating, completing, and failing runs.
    * **Pipeline lifecycle** -- creating, completing, and failing
      pipeline executions.
    * **Chain storage** -- persisting the preprocessing-to-model chain.
    * **Prediction storage** -- saving scalar prediction records and
      their associated dense arrays.
    * **Artifact storage** -- content-addressed persistence of fitted
      Python objects.
    * **Core queries** -- ranking predictions and retrieving single
      records.
    * **Export** -- producing standalone bundles from stored chains.
    """

    # --- Run lifecycle ---

    def begin_run(
        self,
        name: str,
        config: Any,
        datasets: list[dict],
    ) -> str:
        """Create a new run and return its identifier."""
        ...

    def complete_run(self, run_id: str, summary: dict) -> None:
        """Mark a run as completed."""
        ...

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed."""
        ...

    # --- Pipeline lifecycle ---

    def begin_pipeline(
        self,
        run_id: str,
        name: str,
        expanded_config: Any,
        generator_choices: list,
        dataset_name: str,
        dataset_hash: str,
    ) -> str:
        """Register a new pipeline execution under a run."""
        ...

    def complete_pipeline(
        self,
        pipeline_id: str,
        best_val: float,
        best_test: float,
        metric: str,
        duration_ms: int,
    ) -> None:
        """Mark a pipeline as completed."""
        ...

    def fail_pipeline(self, pipeline_id: str, error: str) -> None:
        """Mark a pipeline as failed."""
        ...

    # --- Chain management ---

    def save_chain(
        self,
        pipeline_id: str,
        steps: list[dict],
        model_step_idx: int,
        model_class: str,
        preprocessings: str,
        fold_strategy: str,
        fold_artifacts: dict,
        shared_artifacts: dict,
        branch_path: list[int] | None = None,
        source_index: int | None = None,
    ) -> str:
        """Store a chain and return its identifier."""
        ...

    def get_chain(self, chain_id: str) -> dict | None:
        """Retrieve a chain by identifier."""
        ...

    # --- Prediction storage ---

    def save_prediction(
        self,
        pipeline_id: str,
        chain_id: str,
        dataset_name: str,
        model_name: str,
        model_class: str,
        fold_id: str,
        partition: str,
        val_score: float,
        test_score: float,
        train_score: float,
        metric: str,
        task_type: str,
        n_samples: int,
        n_features: int,
        scores: dict,
        best_params: dict,
        branch_id: int | None,
        branch_name: str | None,
        exclusion_count: int,
        exclusion_rate: float,
        preprocessings: str = "",
        prediction_id: str | None = None,
    ) -> str:
        """Store a prediction record and return its identifier."""
        ...

    def save_prediction_arrays(
        self,
        prediction_id: str,
        y_true: np.ndarray | None,
        y_pred: np.ndarray | None,
        y_proba: np.ndarray | None = None,
        sample_indices: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        """Store prediction arrays."""
        ...

    # --- Artifact storage ---

    def save_artifact(
        self,
        obj: Any,
        operator_class: str,
        artifact_type: str,
        format: str,
    ) -> str:
        """Persist a binary artifact and return its identifier."""
        ...

    def load_artifact(self, artifact_id: str) -> Any:
        """Load a binary artifact by identifier."""
        ...

    # --- Queries ---

    def get_prediction(
        self,
        prediction_id: str,
        load_arrays: bool = False,
    ) -> dict | None:
        """Retrieve a single prediction record."""
        ...

    def top_predictions(
        self,
        n: int,
        metric: str = "val_score",
        ascending: bool = True,
        partition: str = "val",
        dataset_name: str | None = None,
        group_by: str | None = None,
    ) -> pl.DataFrame:
        """Return top-N predictions ranked by a score column."""
        ...

    # --- Export ---

    def export_chain(
        self,
        chain_id: str,
        output_path: Path,
        format: str = "n4a",
    ) -> Path:
        """Export a chain as a standalone bundle."""
        ...
